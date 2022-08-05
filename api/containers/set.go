// Copyright (c) 2022 by Reiform. All Rights Reserved.

package containers

import (
	"reflect"
)

type void struct{}

var setMember void

// UniqueSet a unique set of strings
type UniqueSet[T comparable] struct {
	s map[T]void
}

// NewUniqueSet creates a new unique set
func NewUniqueSet[T comparable]() UniqueSet[T] {
	return UniqueSet[T]{
		s: make(map[T]void),
	}
}

// Union set union with other keys
func (s *UniqueSet[T]) Union(vals ...T) {
	for _, v := range vals {
		s.s[v] = setMember
	}
}

// Vals returns the unique keys
func (s *UniqueSet[T]) Vals() []T {
	keys := make([]T, len(s.s))

	i := 0
	for k := range s.s {
		keys[i] = k
		i++
	}

	return keys
}

// Contains checks if the set contains a particular value
func (s *UniqueSet[T]) Contains(key T) bool {
	_, ok := s.s[key]
	return ok
}

// Merge merges a second unique set into this
func (s *UniqueSet[T]) Merge(other UniqueSet[T]) {
	s.Union(other.Vals()...)
}

// Equals checks set equality
func (s *UniqueSet[T]) Equals(other UniqueSet[T]) bool {
	return reflect.DeepEqual(s.s, other.s)
}
