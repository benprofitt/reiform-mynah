package db

import (
	"errors"
	"fmt"
	"reflect"
	"reiform.com/mynah/model"
	"strconv"
	"unicode"
	"unicode/utf8"
)

//map a struct field to a SQLite type
//returns strings, floats, ints
func sqlORM(s model.Identity, key string) (interface{}, error) {
	r := reflect.ValueOf(s)
	f := reflect.Indirect(r).FieldByName(key)

	first, _ := utf8.DecodeRuneInString(key)

	//verify that the field is exported (important for json serialization)
	if !(unicode.IsUpper(first) && unicode.IsLetter(first)) {
		return nil, errors.New("SQL ORM conversion does not support unexported fields")
	}

	//determine the type
	switch f.Kind() {
	case reflect.Bool:
		return strconv.FormatBool(f.Bool()), nil

	case reflect.Int, reflect.Int8, reflect.Int32, reflect.Int64:
		return f.Int(), nil

	case reflect.Float32, reflect.Float64:
		return f.Float(), nil

	case reflect.String:
		return f.String(), nil

	case reflect.Slice:
		s, jsonErr := serializeJson(f.Interface())
		if jsonErr == nil {
			return *s, nil
		}
		return nil, jsonErr

	case reflect.Map:
		s, jsonErr := serializeJson(f.Interface())
		if jsonErr == nil {
			return *s, nil
		}
		return nil, jsonErr

	default:
		return nil, errors.New(fmt.Sprintf("invalid ORM type conversion or key %s does not exist", key))
	}
}
