package storage

//Defines the interface the storage client must implement
type StorageProvider interface {
}

//local storage client adheres to StorageProvider
type localStorage struct {
}

//external storage client adheres to StorageProvider
type externalStorage struct {
}
