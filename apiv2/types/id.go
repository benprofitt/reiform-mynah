// Copyright (c) 2023 by Reiform. All Rights Reserved.

package types

import "github.com/google/uuid"

type MynahUuid string

// NewMynahUuid creates a new mynah uuid
func NewMynahUuid() MynahUuid {
	return MynahUuid(uuid.NewString())
}
