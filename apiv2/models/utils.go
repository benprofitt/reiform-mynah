// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import "golang.org/x/exp/constraints"

func Min[T constraints.Ordered](a, b T) T {
	if a > b {
		return b
	}
	return a
}

// KeysVals gets the keys and vals for the map in a slice
func KeysVals[K comparable, V any, R any](m map[K]V, transformer func(K, V) *R) []*R {
	res := make([]*R, len(m))

	for k, v := range m {
		res = append(res, transformer(k, v))
	}

	return res
}
