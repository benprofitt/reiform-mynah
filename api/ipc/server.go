// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"sync"
	"syscall"
)

const uuidLength = 36

//ipc provider impl
type ipcServer struct {
	// the socket address
	addr string
	//the socket listener
	listener net.Listener
	//channel of new requests
	messages chan net.Conn
	//wait group for workers
	waitGroup sync.WaitGroup
	//task completion context
	ctx context.Context
	//task completion function
	cancel context.CancelFunc
}

// NewIPCServer creates a new ipc server
func NewIPCServer(sockAddr string) (IPCServer, error) {
	if err := syscall.Unlink(sockAddr); err != nil {
		log.Warnf("failed to unlink ipc socket %s: %s", sockAddr, err)
	}

	//listen to the unix socket
	listener, listenErr := net.Listen("unix", sockAddr)
	if listenErr != nil {
		return nil, fmt.Errorf("failed to listen to ipc socket %s: %s", sockAddr, listenErr)
	}

	//check that the file has been created
	if _, statErr := os.Stat(sockAddr); errors.Is(statErr, os.ErrNotExist) {
		return nil, fmt.Errorf("ipc socket %s does not exist", sockAddr)
	} else if statErr != nil {
		return nil, fmt.Errorf("failed to stat ipc socket %s: %s", sockAddr, statErr)
	}

	s := ipcServer{
		addr:      sockAddr,
		listener:  listener,
		messages:  make(chan net.Conn, 256),
		waitGroup: sync.WaitGroup{},
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return &s, nil
}

// read from a connection
func (s *ipcServer) read(conn net.Conn) ([]byte, error) {
	contents := make([]byte, 0)
	buf := make([]byte, 256)

	for {
		//read the message
		if read, err := conn.Read(buf); err == nil {
			contents = append(contents, buf[:read]...)
		} else if err == io.EOF {
			break
		} else {
			return nil, fmt.Errorf("error reading from ipc connection with addr %s: %s", s.addr, err)
		}
	}

	return contents, nil
}

// Listen for an ipc message
func (s *ipcServer) Listen() ([]byte, error) {
	//accept new connections
	if conn, err := s.listener.Accept(); err == nil {
		return s.read(conn)
	} else {
		return nil, fmt.Errorf("error listening to ipc socket %s: %s", s.addr, err)
	}
}

//write new requests to the handler
func (s *ipcServer) connectionWorker(handler func(userUuid model.MynahUuid, msg []byte)) {
	defer s.waitGroup.Done()

	for {
		select {
		case <-s.ctx.Done():
			return

		case conn := <-s.messages:
			//read from the connection
			if contents, err := s.read(conn); err == nil {
				//read the first 16 bytes
				if len(contents) >= uuidLength {
					s := model.MynahUuid(contents[:uuidLength])
					handler(s, contents[uuidLength:])

				} else {
					log.Warnf("ipc message contained less than %d bytes (%d)", uuidLength, len(contents))
				}
			} else {
				log.Warnf("failed to read from ipc socket with addr %s: %s", s.addr, err)
			}

			if err := conn.Close(); err != nil {
				log.Warnf("failed to close ipc socket connection: %s", err)
			}
		}
	}
}

// ListenMany accepts messages until closed
func (s *ipcServer) ListenMany(handler func(userUuid model.MynahUuid, msg []byte)) {
	//start the connection worker
	go s.connectionWorker(handler)
	s.waitGroup.Add(1)

	for {
		//accept new connections
		if conn, err := s.listener.Accept(); err == nil {
			s.messages <- conn

		} else {
			log.Errorf("error listening to ipc socket %s: %s", s.addr, err)
			return
		}
	}
}

//Close the ipc provider
func (s *ipcServer) Close() {
	//signal goroutines
	s.cancel()
	s.waitGroup.Wait()
	log.Infof("closing ipc server with address: %s", s.addr)

	if err := s.listener.Close(); err != nil {
		log.Warnf("failed to close ipc server with address %s: %s", s.addr, err)
	}
}
