// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/settings"
	"testing"
)

//setup and teardown
func TestMain(m *testing.M) {
	dirPath := "data"

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Fatalf("failed to create directory: %s", dirPath)
	}

	//run tests
	exitVal := m.Run()

	//remove generated
	if err := os.RemoveAll(dirPath); err != nil {
		log.Errorf("failed to clean up after tests: %s", err)
	}

	os.Exit(exitVal)
}

func TestPythonArgs(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.ModulePath = "../../python"

	p := newLocalPythonProvider(mynahSettings)
	defer p.Close()

	_, err := p.InitFunction("mynah_test", "test0")
	require.NoError(t, err)

	_, err = p.InitFunction("mynah_test", "test1")
	require.NoError(t, err)

	_, err = p.InitFunction("mynah_test", "test2")
	require.NoError(t, err)

	_, err = p.InitFunction("mynah_test", "test3")
	require.NoError(t, err)

	_, err = p.InitFunction("mynah_test", "test4")
	require.NoError(t, err)

	arg1_0 := 3
	arg2_0 := 1.2

	res, err := p.localCallFunction("mynah_test", "test0", arg1_0, arg2_0)
	require.NoError(t, err)
	require.NotNil(t, res)
	require.Equal(t, "ab", fmt.Sprintf("%v", res))

	arg1_1 := "c"
	arg2_1 := "d"

	res, err = p.localCallFunction("mynah_test", "test1", arg1_1, arg2_1)
	require.NoError(t, err)
	require.NotNil(t, res)
	require.Equal(t, "cd", fmt.Sprintf("%v", res))

	res, err = p.localCallFunction("mynah_test", "test2")
	require.NoError(t, err)
	require.Nil(t, res)

	arg1_3 := "abc"
	arg2_3 := 5

	res, err = p.localCallFunction("mynah_test", "test3", arg1_3, arg2_3)
	require.NoError(t, err)
	require.NotNil(t, res)
	require.Equal(t, "8", fmt.Sprintf("%v", res))

	_, err = p.localCallFunction("mynah_test", "test4")
	require.Error(t, err)
}
