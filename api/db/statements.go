package db

const createUserTableSQL = `CREATE TABLE user (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "name_first" TEXT,
  "name_last" TEXT,
  "is_admin" TEXT
  );`
