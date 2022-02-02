// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"fmt"
	"reiform.com/mynah/settings"
	"testing"
)

func TestPythonArgs(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.ModulePath = "../../python"

	p := NewPythonProvider(mynahSettings)
	defer p.Close()

	if err := p.InitModule("mynah_test"); err != nil {
		t.Fatalf("failed to init module: %s", err)
	}

	if err := p.InitFunction("mynah_test", "test0"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if err := p.InitFunction("mynah_test", "test1"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if err := p.InitFunction("mynah_test", "test2"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if err := p.InitFunction("mynah_test", "test3"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	if err := p.InitFunction("mynah_test", "test4"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	arg1_0 := 3
	arg2_0 := 1.2

	if res, err := p.CallFunction("mynah_test", "test0", arg1_0, arg2_0); err != nil {
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

	if res, err := p.CallFunction("mynah_test", "test1", arg1_1, arg2_1); err != nil {
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

	if res, err := p.CallFunction("mynah_test", "test2"); err != nil {
		t.Fatalf("failed to call function: %s", err)
	} else {
		//check the result
		if res != nil {
			t.Fatal("result was not nil")
		}
	}

	arg1_3 := "abc"
	arg2_3 := 5

	if res, err := p.CallFunction("mynah_test", "test3", arg1_3, arg2_3); err != nil {
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

	if _, err := p.CallFunction("mynah_test", "test4"); err == nil {
		t.Fatal("python exception test did not produce error")
	}
}
