// Copyright (c) 2022 by Reiform. All Rights Reserved.

package utils

import (
	"errors"
	"fmt"
)

// OneOf returns an error if more than one option selected
func OneOf(options ...string) error {
	found := false
	for _, o := range options {
		if len(o) > 0 {
			if found {
				return errors.New("selected")
			}
			found = true
		}
	}

	if !found {
		return errors.New("empty")
	}
	return nil
}

// IsAllowedTaskType checks that the given type is allowed
func IsAllowedTaskType(given interface{}, allowed ...interface{}) error {
	allowedList := ""
	for _, a := range allowed {
		if given == a {
			return nil
		}
		allowedList = fmt.Sprintf("%s or %s", allowedList, a)
	}
	return fmt.Errorf("%s is not one of %s", given, allowedList)
}
