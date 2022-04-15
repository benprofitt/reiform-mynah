// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"reflect"
	"reiform.com/mynah/model"
)

type void struct{}

var setMember void

// UniqueSet a unique set of strings
type UniqueSet struct {
	//unique set of values
	s map[string]void
}

// NewUniqueSet creates a new unique set
func NewUniqueSet() UniqueSet {
	return UniqueSet{
		s: make(map[string]void),
	}
}

// Union set union with other keys
func (s *UniqueSet) Union(vals ...string) {
	for _, v := range vals {
		s.s[v] = setMember
	}
}

// UuidsUnion set union with other keys
func (s *UniqueSet) UuidsUnion(vals ...model.MynahUuid) {
	for _, v := range vals {
		s.s[string(v)] = setMember
	}
}

// Vals returns the unique keys
func (s *UniqueSet) Vals() []string {
	keys := make([]string, len(s.s))

	i := 0
	for k := range s.s {
		keys[i] = k
		i++
	}

	return keys
}

// UuidVals returns the vals as uuids
func (s *UniqueSet) UuidVals() []model.MynahUuid {
	keys := make([]model.MynahUuid, len(s.s))

	i := 0
	for k := range s.s {
		keys[i] = model.MynahUuid(k)
		i++
	}

	return keys
}

// Contains checks if the set contains a particular value
func (s *UniqueSet) Contains(key string) bool {
	_, ok := s.s[key]
	return ok
}

// Merge merges a second unique set into this
func (s *UniqueSet) Merge(other UniqueSet) {
	s.Union(other.Vals()...)
}

// Equals checks set equality
func (s *UniqueSet) Equals(other UniqueSet) bool {
	return reflect.DeepEqual(s.s, other.s)
}
