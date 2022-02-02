// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"context"
	"fmt"
	//"github.com/google/uuid"
	"io"
	"log"
	"net"
	"os"
	"reiform.com/mynah/settings"
	"sync"
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

//create a new ipc provider fomr settings
func NewIPCProvider(mynahSettings *settings.MynahSettings) (IPCProvider, error) {
	//bind to the socket address
	log.Printf("IPC listening to unix socket %s", mynahSettings.IPCSettings.SocketAddr)

	//remove any existing socket
	if err := os.RemoveAll(mynahSettings.IPCSettings.SocketAddr); err != nil {
		return nil, fmt.Errorf("failed to delete previous socket: %s", err)
	}

	//listen to the unix socket
	listener, listenErr := net.Listen("unix", mynahSettings.IPCSettings.SocketAddr)
	if listenErr != nil {
		return nil, listenErr
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
					log.Printf("error reading from ipc connection: %s", err)
					break
				}
			}

			//read the first 16 bytes
			if len(contents) >= uuidLength {
				s := string(contents[:uuidLength])
				handler(&s, contents[uuidLength:])

			} else {
				log.Printf("ipc message contained less than %d bytes (%d)", uuidLength, len(contents))
			}

			conn.Close()
		}
	}
}

//handle new events
func (s *ipcServer) HandleEvents(handler func(userUuid *string, msg []byte)) {
	//start the connection worker
	go s.connectionWorker(handler)
	s.waitGroup.Add(1)

	for {
		//accept new connections
		if conn, err := s.listener.Accept(); err == nil {
			s.messages <- conn

		} else {
			log.Printf("error listening to socket: %s", err)
			return
		}
	}
}

//Close the ipc provider
func (s *ipcServer) Close() {
	//signal goroutines
	s.cancel()
	s.waitGroup.Wait()
	log.Printf("closing ipc server")
	s.listener.Close()
}
