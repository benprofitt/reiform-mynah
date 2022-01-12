package db

const createUserTableSQL = `CREATE TABLE user (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "org_id" TEXT,
  "name_first" TEXT,
  "name_last" TEXT,
  "is_admin" TEXT,
  "created_by" TEXT
  );`

const createUserSQL = `INSERT INTO
  user(uuid, org_id, name_first, name_last, is_admin, created_by)
  VALUES (?, ?, ?, ?, ?, ?)`

//Note: created_by, org_id is never updated after creation
const updateUserSQL = `UPDATE user SET
  name_first = ?,
  name_last = ?,
  is_admin = ?,
  WHERE uuid = ? AND org_id = ?`

//TODO only list in org
const listUsersSQL = ``

//TODO only list in org
const listProjectSQL = ``
