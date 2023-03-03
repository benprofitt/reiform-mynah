// Copyright (c) 2023 by Reiform. All Rights Reserved.

package db

import (
	"context"
	"reiform.com/mynah-api/services/log"
	"xorm.io/xorm"
)

type engineContext interface {
	newTransaction() *xorm.Session
	getInterface() xorm.Interface
}

type xormEngine struct {
	eng *xorm.Engine
}

type xormSession struct {
	sess *xorm.Session
}

func (e *xormEngine) newTransaction() *xorm.Session {
	return e.eng.NewSession()
}

func (s *xormSession) newTransaction() *xorm.Session {
	return s.sess
}

func (e *xormEngine) getInterface() xorm.Interface {
	return e.eng
}

func (s *xormSession) getInterface() xorm.Interface {
	return s.sess
}

// Context represents a db context
type Context struct {
	context.Context
	engine engineContext
}

// NewContext creates a new context
func NewContext() *Context {
	return &Context{
		context.Background(),
		&xormEngine{
			coreEngine,
		},
	}
}

// Engine gets the engine for the current context
func (c *Context) Engine() xorm.Interface {
	return c.engine.getInterface()
}

// NewTransaction executes a transaction for the duration of the handler with rollback
func (c *Context) NewTransaction(handler func(*Context) error) error {
	sess := c.engine.newTransaction()
	defer func(s *xorm.Session) {
		if err := s.Close(); err != nil {
			log.Error("failed to close db session after transaction: %s", err)
		}
	}(sess)

	prevEngine := c.engine
	defer func(e engineContext) {
		// swap the previous engine in
		c.engine = e
	}(prevEngine)

	// begin transaction
	if err := sess.Begin(); err != nil {
		return err
	}

	// set the new engine for the context
	c.engine = &xormSession{
		sess,
	}

	if err := handler(c); err != nil {
		if rbErr := sess.Rollback(); rbErr != nil {
			log.Error("failed to rollback transaction: %s", rbErr)
		}
		// return the original err
		return err
	}

	// commit the transaction
	return sess.Commit()
}
