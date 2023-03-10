package cmap

import "testing"

type Animal struct {
	name string
}

type Key struct {
	name string
	age  uint
}

func TestMapCreation(t *testing.T) {
	m := New[Animal, string]()
	if m.shards == nil {
		t.Error("map is null.")
	}

	if m.Count() != 0 {
		t.Error("new map should be empty.")
	}
}

func TestInsert(t *testing.T) {
	m := New[string, Animal]()
	elephant := Animal{"elephant"}
	monkey := Animal{"monkey"}
	m.Set("elephant", elephant)
	m.Set("monkey", monkey)

	if m.Count() != 2 {
		t.Error("map should contain exactly two elements.")
	}
}

func TestGet(t *testing.T) {
	m := New[string, Animal]()

	// Get a missing element.
	val, ok := m.Get("Money")

	if ok == true {
		t.Error("ok should be false when item is missing from map.")
	}

	if (val != Animal{}) {
		t.Error("Missing values should return as null.")
	}

	elephant := Animal{"elephant"}
	m.Set("elephant", elephant)

	// Retrieve inserted element.
	elephant, ok = m.Get("elephant")
	if ok == false {
		t.Error("ok should be true for item stored within the map.")
	}

	if elephant.name != "elephant" {
		t.Error("item was modified.")
	}
}

func TestGetWithStructKey(t *testing.T) {
	m := New[Key, Animal]()

	// Get a missing element.
	val, ok := m.Get(Key{name: "John", age: 15})

	if ok == true {
		t.Error("ok should be false when item is missing from map.")
	}

	if (val != Animal{}) {
		t.Error("Missing values should return as null.")
	}

	elephant := Animal{"elephant"}
	user := Key{name: "Mark", age: 42}
	m.Set(user, elephant)

	// Retrieve inserted element.
	elephant, ok = m.Get(user)
	if ok == false {
		t.Error("ok should be true for item stored within the map.")
	}

	if elephant.name != "elephant" {
		t.Error("item was modified.")
	}
}
