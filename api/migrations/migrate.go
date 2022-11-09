// Copyright (c) 2022 by Reiform. All Rights Reserved.

package migrations

import (
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"sort"
	"xorm.io/xorm"
)

type Migration interface {
	// Migrate performs the migration
	Migrate(*xorm.Session) error
	// Id gets the migration id
	Id() model.MynahUuid
}

type MigrationRecord struct {
	Id          int64           `xorm:"pk autoincr"`
	MigrationId model.MynahUuid `xorm:"INDEX UNIQUE"`
}

var migrations = make(map[model.MynahUuid]Migration)

// check if a migration has already been applied
func migrationApplied(x *xorm.Session, m Migration) (bool, error) {
	return x.Get(&MigrationRecord{MigrationId: m.Id()})
}

// apply a migration
func applyMigration(x *xorm.Session, m Migration) error {
	if err := m.Migrate(x); err != nil {
		return err
	}

	_, err := x.Insert(&MigrationRecord{
		MigrationId: m.Id(),
	})
	return err
}

// Migrate the database as needed
func Migrate(engine *xorm.Engine) error {
	if err := engine.Sync(new(MigrationRecord)); err != nil {
		return err
	}

	// sort the migrations by id
	sortedMigrations := make([]Migration, 0, len(migrations))

	for _, m := range migrations {
		sortedMigrations = append(sortedMigrations, m)
	}

	sort.Slice(sortedMigrations, func(i, j int) bool {
		return sortedMigrations[i].Id() < sortedMigrations[j].Id()
	})

	sess := engine.NewSession()
	defer func(s *xorm.Session) {
		if err := s.Close(); err != nil {
			log.Errorf("Failed to close db session after migration transaction: %s", err)
		}
	}(sess)

	rollback := func(s *xorm.Session) {
		if err := s.Rollback(); err != nil {
			log.Errorf("failed to rollback migration transaction")
		}
	}

	for _, migration := range sortedMigrations {

		if applied, err := migrationApplied(sess, migration); err != nil {
			rollback(sess)
			return err
		} else if !applied {
			// apply the migration
			if err := applyMigration(sess, migration); err != nil {
				rollback(sess)
				log.Infof("failed to apply migration[%s]: %s", migration.Id(), err)
				return err
			}

			log.Infof("applied migration[%s]", migration.Id())
		}

		if err := sess.Commit(); err != nil {
			rollback(sess)
			return err
		}
	}

	return nil
}

func registerMigration(m Migration) {
	migrations[m.Id()] = m
}
