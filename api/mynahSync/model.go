// Copyright (c) 2022 by Reiform. All Rights Reserved.

package mynahSync

import "reiform.com/mynah/model"

// MynahSyncLock defines a mynah lock
type MynahSyncLock interface {
	// Unlock the lock
	Unlock()
}

// MynahSyncLockSet defines a collection of simultaneous locks
type MynahSyncLockSet map[model.MynahUuid]MynahSyncLock

// MynahSyncProvider defines a synchronization provider
type MynahSyncProvider interface {
	// Lock a resource by uuid
	Lock(model.MynahUuid) (MynahSyncLock, error)
	// LockMany locks a set of resources (blocking)
	LockMany(model.MynahUuidList) (MynahSyncLockSet, error)
}
