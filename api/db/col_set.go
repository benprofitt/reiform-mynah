// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"fmt"
	"reiform.com/mynah/model"
)

type void struct{}

var colMember void

// MynahDBColumns maintains column dependencies on update
type MynahDBColumns interface {
	// cols gets the column name and any dependent columns
	cols() ([]string, error)
	// empty returns true if no cols added
	empty() bool
	// contains returns true if the set contains a given key
	contains(model.MynahColName) bool
	// containsRestricted returns true if a non updatable column is included
	containsRestricted() (bool, model.NonUpdatableColName)
	// add adds a key
	add(model.MynahColName)
}

type localColSet struct {
	// whether the col set is somehow invalid
	invalid error
	// the cols to reference
	colSet map[model.MynahColName]void
}

// Cols gets the cols from this set
func (c localColSet) cols() (cols []string, err error) {
	if c.invalid != nil {
		return cols, c.invalid
	}

	for col := range c.colSet {
		cols = append(cols, string(col))
	}
	return cols, nil
}

// Empty returns true if no cols added
func (c localColSet) empty() bool {
	return len(c.colSet) == 0
}

// Contains returns true if the set contains a given key
func (c localColSet) contains(col model.MynahColName) bool {
	_, ok := c.colSet[col]
	return ok
}

// Contains returns true if the set contains a given key
func (c localColSet) containsRestricted() (bool, model.NonUpdatableColName) {
	for col := range c.colSet {
		if _, ok := model.MynahNonUpdatableDatabaseColumns[model.NonUpdatableColName(col)]; ok {
			return true, model.NonUpdatableColName(col)
		}
	}
	return false, ""
}

// Add adds a key
func (c localColSet) add(col model.MynahColName) {
	c.colSet[col] = colMember
}

// NewMynahDBColumns creates a new column update set
func NewMynahDBColumns(cols ...model.MynahColName) MynahDBColumns {
	colSet := make(map[model.MynahColName]void)

	for _, c := range cols {
		colSet[c] = colMember

		if dependent, ok := model.MynahDatabaseColumns[c]; ok {
			for _, d := range dependent {
				colSet[d] = colMember
			}

		} else {
			return &localColSet{
				invalid: fmt.Errorf("untracked database column: %s", c),
				colSet:  make(map[model.MynahColName]void),
			}
		}
	}

	return &localColSet{
		invalid: nil,
		colSet:  colSet,
	}
}

// NewMynahDBColumnsNoDeps creates a set but does not include dependent columns
func NewMynahDBColumnsNoDeps(cols ...model.MynahColName) MynahDBColumns {
	set := localColSet{
		invalid: nil,
		colSet:  make(map[model.MynahColName]void),
	}

	for _, c := range cols {
		set.colSet[c] = colMember
	}
	return &set
}
