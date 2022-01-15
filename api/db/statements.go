package db

const createUserTableSQL = `CREATE TABLE user (
  "Uuid" TEXT NOT NULL PRIMARY KEY,
  "OrgId" TEXT,
  "NameFirst" TEXT,
  "NameLast" TEXT,
  "IsAdmin" TEXT,
  "CreatedBy" TEXT
  );`

const createProjectTableSQL = `CREATE TABLE project (
  "Uuid" TEXT NOT NULL PRIMARY KEY,
  "OrgId" TEXT,
  "UserPermissions" TEXT,
  "ProjectName" TEXT
  );`

const createFileTableSQL = `CREATE TABLE file (
  "Uuid" TEXT NOT NULL PRIMARY KEY,
  "OrgId" TEXT,
  "OwnerUuid" TEXT,
  "Name" TEXT,
  "Location" TEXT,
  "Path" TEXT
  );`

//Note: we can't filter by OrgId since it isn't known
//when authenticating a user -- we rely on the uniqueness of Uuids
const getUserSQL = `SELECT
  Uuid,
  OrgId,
  NameFirst,
  NameLast,
  IsAdmin,
  CreatedBy
  FROM user
  WHERE Uuid = ?`

//filter by both Uuid and org id (although both _should_ be unique)
const getProjectSQL = `SELECT
  Uuid,
  OrgId,
  UserPermissions,
  ProjectName
  FROM project
  WHERE Uuid = ? AND OrgId = ?`

//filter by both Uuid and org id (although both _should_ be unique)
const getFileSQL = `SELECT
  Uuid
  OrgId,
  OwnerUuid,
  Name,
  Location,
  Path
  FROM file
  WHERE Uuid = ? AND OrgId = ?`

const listUsersSQL = `SELECT
  Uuid,
  OrgId,
  NameFirst,
  NameLast,
  IsAdmin,
  CreatedBy
  FROM user
  WHERE OrgId = ?`

const listProjectsSQL = `SELECT
  Uuid,
  OrgId,
  UserPermissions,
  ProjectName
  FROM project
  WHERE OrgId = ?`

const listFilesSQL = `SELECT
  Uuid
  OrgId,
  OwnerUuid,
  Name,
  Location,
  Path
  FROM file
  WHERE OrgId = ?`

const createUserSQL = `INSERT INTO
  user(Uuid, OrgId, NameFirst, NameLast, IsAdmin, CreatedBy)
  VALUES (?, ?, ?, ?, ?, ?)`

const createProjectSQL = `INSERT INTO
  project(Uuid, OrgId, UserPermissions, ProjectName)
  VALUES (?, ?, ?, ?)`

const createFileSQL = `INSERT INTO
  file(Uuid, OrgId, OwnerUuid, Name, Location, Path)
  VALUES (?, ?, ?, ?, ?, ?)`

const deleteUserSQL = `DELETE FROM user
  WHERE Uuid = ? AND OrgId = ?`

const deleteProjectSQL = `DELETE FROM project
  WHERE Uuid = ? AND OrgId = ?`

const deleteFileSQL = `DELETE FROM file
  WHERE Uuid = ? AND OrgId = ?`
