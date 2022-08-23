// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

type MynahUuid string

type MynahUuidList []MynahUuid

// Len returns 1 (sort interface)
func (ids MynahUuidList) Len() int {
	return len(ids)
}

// Less compares two ids
func (ids MynahUuidList) Less(i, j int) bool {
	return string(ids[i]) < string(ids[j])
}

// Swap swaps two ids
func (ids MynahUuidList) Swap(i, j int) {
	ids[i], ids[j] = ids[j], ids[i]
}
