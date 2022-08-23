// Copyright (c) 2022 by Reiform. All Rights Reserved.

package mynahSync

import (
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"sort"
	"sync"
)

// localSyncProvider defines a mynahSync provider for this server only
type localSyncProvider struct {
	// the locks currently held
	locks map[model.MynahUuid]*sync.Mutex
	//the lock on the entire map
	accessLock sync.Mutex
}

// there should only ever be one source of locks locally
var provider = localSyncProvider{
	locks:      make(map[model.MynahUuid]*sync.Mutex),
	accessLock: sync.Mutex{},
}

// ConfigureSync configures synchronization
func ConfigureSync(mynahSettings *settings.MynahSettings) error {
	log.Infof("configured for local resource synchronization")
	return nil
}

// GetSyncProvider gets the configured synchronization provider
func GetSyncProvider() MynahSyncProvider {
	return &provider
}

// Lock a resource by uuid
func (p *localSyncProvider) Lock(uuid model.MynahUuid) (MynahSyncLock, error) {
	p.accessLock.Lock()
	defer p.accessLock.Unlock()
	//Note: currently we don't ever delete locks. Maybe we should
	if _, exists := p.locks[uuid]; !exists {
		p.locks[uuid] = &sync.Mutex{}
	}
	p.locks[uuid].Lock()
	return p.locks[uuid], nil
}

// LockMany locks a set of resources (blocking)
func (p *localSyncProvider) LockMany(uuids model.MynahUuidList) (MynahSyncLockSet, error) {
	// Note: we first need to sort the uuids to avoid a deadlock. For example, at timestep 1 say process 1 locks
	// resource A and process 2 locks resource B. Then, at timestep 2, if each process tries to acquire the other lock
	// we enter a deadlock
	sort.Sort(uuids)

	locks := make(MynahSyncLockSet)

	//lock in order
	for _, uuid := range uuids {
		l, err := p.Lock(uuid)
		if err != nil {
			//unlock all
			locks.CheckUnlocked()
			return nil, err
		}
		locks[uuid] = l
	}

	return locks, nil
}

// CheckUnlocked all the locks but reports a warning if some have not already been unlocked
func (s MynahSyncLockSet) CheckUnlocked() {
	if len(s) > 0 {
		// likely a bug
		log.Errorf("CheckUnlocked(): lock set has %d locks remaining", len(s))
	}
	for _, lock := range s {
		lock.Unlock()
	}
}

// Remove a lock from the set
func (s MynahSyncLockSet) Remove(id model.MynahUuid) {
	delete(s, id)
}
