// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

type void struct{}

var setMember void

// UniqueSet a unique set of strings
type UniqueSet struct {
	//unique set of values
	s map[string]void
}

// NewUniqueSet creates a new unique set
func NewUniqueSet() *UniqueSet {
	return &UniqueSet{
		s: make(map[string]void),
	}
}

// Union set union with other keys
func (s *UniqueSet) Union(vals ...string) {
	for _, v := range vals {
		s.s[v] = setMember
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
