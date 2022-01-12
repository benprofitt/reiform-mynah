package db

const createUserTableSQL = `CREATE TABLE user (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "org_id" TEXT,
  "name_first" TEXT,
  "name_last" TEXT,
  "is_admin" TEXT,
  "created_by" TEXT
  );`

const createProjectTableSQL = ``

const getUserSQL = `SELECT
  uuid,
  org_id,
  name_first,
  name_last,
  is_admin,
  created_by
  FROM user
  WHERE uuid = ? AND org_id = ?`

const getProjectSQL = ``

const listUsersSQL = `SELECT
  uuid,
  org_id,
  name_first,
  name_last,
  is_admin,
  created_by
  FROM user
  WHERE org_id = ?`

//TODO only list in org
const listProjectSQL = ``

const createUserSQL = `INSERT INTO
  user(uuid, org_id, name_first, name_last, is_admin, created_by)
  VALUES (?, ?, ?, ?, ?, ?)`

const createProjectSQL = ``

//Note: created_by, org_id is never updated after creation
const updateUserSQL = `UPDATE user SET
  name_first = ?,
  name_last = ?,
  is_admin = ?,
  WHERE uuid = ? AND org_id = ?`

const updateProjectSQL = ``

const deleteUserSQL = `DELETE FROM user
  WHERE uuid = ? AND org_id = ?`

const deleteProjectSQL = ``
