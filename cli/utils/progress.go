// Copyright (c) 2022 by Reiform. All Rights Reserved.

package utils

import (
	"github.com/vbauerster/mpb/v7"
	"github.com/vbauerster/mpb/v7/decor"
)

// TaskProgress indicates task progress to the user
type TaskProgress interface {
	// Increment signals that a unit of the task has completed
	Increment()
	// Complete signals that the task has completed
	Complete()
}

type cliTaskProgressBar struct {
	progress *mpb.Progress
	//the bar
	bar *mpb.Bar
}

// Increment indicates that another task unit has completed
func (c cliTaskProgressBar) Increment() {
	c.bar.Increment()
}

// Complete marks the task as completed
func (c cliTaskProgressBar) Complete() {
	c.progress.Wait()
}

// NewCLITaskProgressBar creates a progress bar
func NewCLITaskProgressBar(title string, totalUnits int64) TaskProgress {
	progress := mpb.New()
	return &cliTaskProgressBar{
		progress: progress,
		bar: progress.New(totalUnits,
			// BarFillerBuilder with custom style
			mpb.BarStyle().Lbound("╢").Filler("▌").Tip("▌").Padding("░").Rbound("╟"),
			mpb.PrependDecorators(
				decor.Name(title, decor.WC{W: len(title) + 1, C: decor.DidentRight}),
				decor.OnComplete(
					decor.AverageETA(decor.ET_STYLE_GO, decor.WC{W: 4}), "done",
				),
			),
			mpb.AppendDecorators(decor.Percentage()),
		),
	}
}
