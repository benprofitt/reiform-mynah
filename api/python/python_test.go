// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"fmt"
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

	if _, err := p.InitFunction("mynah_test", "test0"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if _, err := p.InitFunction("mynah_test", "test1"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if _, err := p.InitFunction("mynah_test", "test2"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if _, err := p.InitFunction("mynah_test", "test3"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if _, err := p.InitFunction("mynah_test", "test4"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	arg1_0 := 3
	arg2_0 := 1.2

	if res, err := p.localCallFunction("mynah_test", "test0", arg1_0, arg2_0); err != nil {
		t.Fatalf("failed to call function: %s", err)
	} else {
		//check the result
		if res == nil {
			t.Fatal("result was nil")
			return
		}
		if s := fmt.Sprintf("%v", res); s != "ab" {
			t.Fatalf("%s != ab", s)
		}
	}

	arg1_1 := "c"
	arg2_1 := "d"

	if res, err := p.localCallFunction("mynah_test", "test1", arg1_1, arg2_1); err != nil {
		t.Fatalf("failed to call function: %s", err)
	} else {
		//check the result
		if res == nil {
			t.Fatal("result was nil")
		}
		if s := fmt.Sprintf("%v", res); s != "cd" {
			t.Fatalf("%s != cd", s)
		}
	}

	if res, err := p.localCallFunction("mynah_test", "test2"); err != nil {
		t.Fatalf("failed to call function: %s", err)
	} else {
		//check the result
		if res != nil {
			t.Fatal("result was not nil")
		}
	}

	arg1_3 := "abc"
	arg2_3 := 5

	if res, err := p.localCallFunction("mynah_test", "test3", arg1_3, arg2_3); err != nil {
		t.Fatalf("failed to call function: %s", err)
	} else {
		//check the result
		if res == nil {
			t.Fatal("result was nil")
		}
		if s := fmt.Sprintf("%v", res); s != "8" {
			t.Fatalf("%s != 8", s)
		}
	}

	if _, err := p.localCallFunction("mynah_test", "test4"); err == nil {
		t.Fatal("python exception test did not produce error")
	}
}
