// Copyright (c) 2022 by Reiform. All Rights Reserved.

package engine

import (
	"reiform.com/mynah/log"
	"xorm.io/xorm"
)

type Engine interface {
	// GetEngine gets the xorm engine for executing queries
	GetEngine() xorm.Interface
	// NewTransaction executes a transaction for the duration of the handler with rollback
	NewTransaction(func(Engine) error) error
	// CloseEngine closes the engine
	CloseEngine() error
}

type xormEngine struct {
	*xorm.Engine
}

type xormSession struct {
	*xorm.Session
}

// NewEngine creates a new xorm engine
func NewEngine(engine *xorm.Engine) Engine {
	return &xormEngine{
		engine,
	}
}

// GetEngine gets the xorm engine for executing queries
func (e *xormEngine) GetEngine() xorm.Interface {
	return e
}

// NewTransaction executes a transaction for the duration of the handler with rollback
func (e *xormEngine) NewTransaction(handler func(Engine) error) error {
	sess := e.NewSession()
	defer func(s *xorm.Session) {
		if err := s.Close(); err != nil {
			log.Errorf("failed to close db session after transaction: %s", err)
		}
	}(sess)

	// begin transaction
	if err := sess.Begin(); err != nil {
		return err
	}

	if err := handler(&xormSession{sess}); err != nil {
		if rbErr := sess.Rollback(); rbErr != nil {
			log.Errorf("failed to rollback transaction: %s", rbErr)
		}
		// return the original err
		return err
	}
	// commit the transaction
	return sess.Commit()
}

// CloseEngine the engine
func (e *xormEngine) CloseEngine() error {
	return e.Close()
}

// GetEngine gets the xorm engine for executing queries
func (e *xormSession) GetEngine() xorm.Interface {
	return e
}

// NewTransaction executes a transaction for the duration of the handler with rollback
func (e *xormSession) NewTransaction(handler func(Engine) error) error {
	log.Warnf("Transaction() called on transaction already started - possible bug")
	return handler(e)
}

// CloseEngine the engine
func (e *xormSession) CloseEngine() error {
	// session should be closed in NewTransaction
	return nil
}
