package db

const createUserTableSQL = `CREATE TABLE user (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "org_id" TEXT,
  "name_first" TEXT,
  "name_last" TEXT,
  "is_admin" TEXT,
  "created_by" TEXT
  );`

const createProjectTableSQL = `CREATE TABLE project (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "org_id" TEXT,
  "user_permissions" TEXT,
  "project_name" TEXT
  );`

const createFileTableSQL = `CREATE TABLE file (
  "uuid" TEXT NOT NULL PRIMARY KEY,
  "org_id" TEXT,
  "owner_uuid" TEXT,
  "name" TEXT,
  "location" TEXT,
  "path" TEXT
  );`

//Note: we can't filter by org_id since it isn't known
//when authenticating a user -- we rely on the uniqueness of uuids
const getUserSQL = `SELECT
  uuid,
  org_id,
  name_first,
  name_last,
  is_admin,
  created_by
  FROM user
  WHERE uuid = ?`

//filter by both uuid and org id (although both _should_ be unique)
const getProjectSQL = `SELECT
  uuid,
  org_id,
  user_permissions,
  project_name
  FROM project
  WHERE uuid = ? AND org_id = ?`

//filter by both uuid and org id (although both _should_ be unique)
const getFileSQL = `SELECT
  uuid
  org_id,
  owner_uuid,
  name,
  location,
  path
  FROM file
  WHERE uuid = ? AND org_id = ?`

const listUsersSQL = `SELECT
  uuid,
  org_id,
  name_first,
  name_last,
  is_admin,
  created_by
  FROM user
  WHERE org_id = ?`

const listProjectsSQL = `SELECT
  uuid,
  org_id,
  user_permissions,
  project_name
  FROM project
  WHERE org_id = ?`

const listFilesSQL = `SELECT
  uuid
  org_id,
  owner_uuid,
  name,
  location,
  path
  FROM file
  WHERE org_id = ?`

const createUserSQL = `INSERT INTO
  user(uuid, org_id, name_first, name_last, is_admin, created_by)
  VALUES (?, ?, ?, ?, ?, ?)`

const createProjectSQL = `INSERT INTO
  project(uuid, org_id, user_permissions, project_name)
  VALUES (?, ?, ?, ?)`

const createFileSQL = `INSERT INTO
  file(uuid, org_id, owner_uuid, name, location, path)
  VALUES (?, ?, ?, ?, ?, ?)`

//Note: created_by, org_id is never updated after creation
const updateUserSQL = `UPDATE user SET
  name_first = ?,
  name_last = ?,
  is_admin = ?,
  WHERE uuid = ? AND org_id = ?`

const updateProjectSQL = `UPDATE project SET
  user_permissions = ?,
  project_name = ?
  WHERE uuid = ? AND org_id = ?`

const updateFileSQL = `UPDATE file SET
  owner_uuid = ?,
  name = ?,
  location = ?,
  path = ?
  WHERE uuid = ? AND org_id = ?`

const deleteUserSQL = `DELETE FROM user
  WHERE uuid = ? AND org_id = ?`

const deleteProjectSQL = `DELETE FROM project
  WHERE uuid = ? AND org_id = ?`

const deleteFileSQL = `DELETE FROM file
  WHERE uuid = ? AND org_id = ?`
