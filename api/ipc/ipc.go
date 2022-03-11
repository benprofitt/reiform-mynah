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
	"reiform.com/mynah/settings"
	"sync"
	"syscall"
)

const uuidLength = 36

//ipc provider impl
type ipcServer struct {
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

// NewIPCProvider create a new ipc provider from settings
func NewIPCProvider(mynahSettings *settings.MynahSettings) (IPCProvider, error) {
	//bind to the socket address
	log.Infof("IPC listening to unix socket %s", mynahSettings.IPCSettings.SocketAddr)

	if err := syscall.Unlink(mynahSettings.IPCSettings.SocketAddr); err != nil {
		log.Warnf("failed to unlink ipc socket %s: %s", mynahSettings.IPCSettings.SocketAddr, err)
	}

	//listen to the unix socket
	listener, listenErr := net.Listen("unix", mynahSettings.IPCSettings.SocketAddr)
	if listenErr != nil {
		return nil, listenErr
	}

	//check that the file has been created
	if _, statErr := os.Stat(mynahSettings.IPCSettings.SocketAddr); errors.Is(statErr, os.ErrNotExist) {
		return nil, fmt.Errorf("ipc socket %s does not exist", mynahSettings.IPCSettings.SocketAddr)
	} else if statErr != nil {
		return nil, fmt.Errorf("failed to stat ipc socket: %s", statErr)
	}

	s := ipcServer{
		listener:  listener,
		messages:  make(chan net.Conn, 256),
		waitGroup: sync.WaitGroup{},
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return &s, nil
}

//write new requests to the handler
func (s *ipcServer) connectionWorker(handler func(userUuid *string, msg []byte)) {
	defer s.waitGroup.Done()

	for {
		select {
		case <-s.ctx.Done():
			return

		case conn := <-s.messages:
			//read from the connection
			contents := make([]byte, 0)
			buf := make([]byte, 256)

			for {
				//read the message
				if read, err := conn.Read(buf); err == nil {
					contents = append(contents, buf[:read]...)
				} else if err == io.EOF {
					break
				} else {
					log.Errorf("error reading from ipc connection: %s", err)
					break
				}
			}

			//read the first 16 bytes
			if len(contents) >= uuidLength {
				s := string(contents[:uuidLength])
				handler(&s, contents[uuidLength:])

			} else {
				log.Warnf("ipc message contained less than %d bytes (%d)", uuidLength, len(contents))
			}

			if err := conn.Close(); err != nil {
				log.Warnf("failed to close ipc socket connection: %s", err)
			}
		}
	}
}

// HandleEvents handle new events
func (s *ipcServer) HandleEvents(handler func(userUuid *string, msg []byte)) {
	//start the connection worker
	go s.connectionWorker(handler)
	s.waitGroup.Add(1)

	for {
		//accept new connections
		if conn, err := s.listener.Accept(); err == nil {
			s.messages <- conn

		} else {
			log.Warnf("error listening to socket: %s", err)
			return
		}
	}
}

//Close the ipc provider
func (s *ipcServer) Close() {
	//signal goroutines
	s.cancel()
	s.waitGroup.Wait()
	log.Infof("closing ipc server")

	if err := s.listener.Close(); err != nil {
		log.Warnf("failed to close ipc server: %s", err)
	}
}
