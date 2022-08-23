// Copyright (c) 2022 by Reiform. All Rights Reserved.

package mynahSync

import (
	"github.com/stretchr/testify/require"
	"testing"
)

//Test the behavior of authentication
func TestLock(t *testing.T) {
	p := GetSyncProvider()

	l, err := p.Lock("test")
	require.NoError(t, err)
	require.NotNil(t, l)

	// verify that the lock is held
	require.False(t, provider.locks["test"].TryLock())

	l.Unlock()

	// verify that the lock is released
	require.True(t, provider.locks["test"].TryLock())
}
