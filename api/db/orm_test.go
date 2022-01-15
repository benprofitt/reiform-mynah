package db

import (
	"fmt"
	"reflect"
	"reiform.com/mynah/model"
	"testing"
)

type NestedStruct struct {
	Val int `json:"val"`
}

type ormTestType struct {
	Uuid       string
	OrgId      string
	Str        string
	I          int64
	F          float64
	B          bool
	M          map[string]int
	S          []string
	unExported int
	Nest       NestedStruct
}

func (o *ormTestType) GetUuid() string {
	return o.Uuid
}

func (o *ormTestType) GetOrgId() string {
	return o.OrgId
}

//verify that a basic struct can be converted to SQL friendly types using reflection
func TestReflectionORMMapper(t *testing.T) {
	s := ormTestType{
		Str: "string",
		I:   9,
		F:   1.2,
		B:   true,
		M:   make(map[string]int),
		S:   make([]string, 0, 2),
	}

	s.M["key"] = 5
	s.S = append(s.S, "str1")
	s.S = append(s.S, "str2")

	if str, err := sqlORM(&s, "Str"); err == nil {
		if sRep := fmt.Sprintf("%v", str); sRep != s.Str {
			t.Errorf("%s != %s", sRep, s.Str)
			return
		}
	} else {
		t.Errorf("string reflection error %s", err)
		return
	}

	if i, err := sqlORM(&s, "I"); err == nil {
		expected := fmt.Sprintf("%d", s.I)
		if sRep := fmt.Sprintf("%v", i); sRep != expected {
			t.Errorf("%s != %s", sRep, expected)
			return
		}
	} else {
		t.Errorf("int reflection error %s", err)
		return
	}

	if f, err := sqlORM(&s, "F"); err == nil {
		expected := fmt.Sprintf("%.1f", s.F)
		if sRep := fmt.Sprintf("%v", f); sRep != expected {
			t.Errorf("%s != %s", sRep, expected)
			return
		}
	} else {
		t.Errorf("float reflection error %s", err)
		return
	}

	if b, err := sqlORM(&s, "B"); err == nil {
		expected := fmt.Sprintf("%v", s.B)
		if sRep := fmt.Sprintf("%v", b); sRep != expected {
			t.Errorf("%s != %s", sRep, expected)
			return
		}
	} else {
		t.Errorf("float reflection error %s", err)
		return
	}

	if m, err := sqlORM(&s, "M"); err == nil {
		expected, _ := serializeJson(s.M)
		if sRep := fmt.Sprintf("%v", m); sRep != *expected {
			t.Errorf("%s != %s", sRep, *expected)
			return
		}
	} else {
		t.Errorf("map reflection error %s", err)
		return
	}

	if sl, err := sqlORM(&s, "S"); err == nil {
		expected, _ := serializeJson(s.S)
		if sRep := fmt.Sprintf("%v", sl); sRep != *expected {
			t.Errorf("%s != %s", sRep, *expected)
			return
		}
	} else {
		t.Errorf("slice reflection error %s", err)
		return
	}

	//nested struct type
	if _, err := sqlORM(&s, "Nest"); err == nil {
		t.Error("no error for nested type")
	}

	//test a nonexistant key
	if _, err := sqlORM(&s, "NonKey"); err == nil {
		t.Error("no error for non key")
	}

	//test an unexported field
	if _, err := sqlORM(&s, "unExported"); err == nil {
		t.Error("no error for unexported key")
	}
}

//test a mynah type
func testMynahType(mynahType model.Identity) error {
	v := reflect.ValueOf(mynahType).Elem()
	for i := 0; i < v.NumField(); i++ {
		if _, err := sqlORM(mynahType, v.Type().Field(i).Name); err != nil {
			return err
		}
	}
	return nil
}

//check that mynah types can be converted
func TestReflectionORMMynahTypes(t *testing.T) {
	if err := testMynahType(&model.MynahUser{}); err != nil {
		t.Errorf("error for type model.MynahUser: %s", err)
	}

	if err := testMynahType(&model.MynahProject{}); err != nil {
		t.Errorf("error for type model.MynahProject %s", err)
	}

	if err := testMynahType(&model.MynahFile{}); err != nil {
		t.Errorf("error for type model.MynahFile %s", err)
	}
}
