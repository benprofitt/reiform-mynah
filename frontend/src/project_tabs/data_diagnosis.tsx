import React from "react";
import { Menu } from "@headlessui/react";
// import { ChevronDownIcon } from '@heroicons/react/solid'

export default function DataDiagnosis(): JSX.Element {
  const [ad_checked, setAdChecked] = React.useState(true);
  const [ms_checked, setMsChecked] = React.useState(false);
  const [sd_checked, setSdChecked] = React.useState(false);
  const [sr_checked, setSrChecked] = React.useState(false);
  const [fc_checked, setFcChecked] = React.useState(false);
  const [sii_checked, setSiiChecked] = React.useState(false);
  const [le_checked, setLeChecked] = React.useState(false);
  const [lcv_checked, setLcvChecked] = React.useState(false);
  const [li_checked, setLiChecked] = React.useState(false);
  const [b_checked, setBChecked] = React.useState(false);
  return (
    <div>
      <div className="ml-3 mt-3 text-xl">
        Choose the issues you would like us to search for while we analyze your
        dataset:
      </div>
      <div className="mt-3 ml-5 border-l-2 border-y-2 border-black h-3"></div>
      <div className="ml-10 mt-3 text-lg">
        <label>
          <input
            className="mr-2"
            type="radio"
            name="radio"
            defaultChecked={ad_checked}
            onChange={() => {
              setAdChecked(true);
              setMsChecked(false);
              setSdChecked(false);
              setFcChecked(false);
              setSiiChecked(false);
              setSrChecked(false);
              setLeChecked(false);
              setLcvChecked(false);
              setLiChecked(false);
              setLiChecked(false);
              setBChecked(false);
            }}
          />
          Auto-Diagnosis (Recommended)
        </label>
        <div className="mt-3 border-l-2 border-y-2 border-black h-3"></div>
        <div className="mt-3">
          <label>
            <input
              className="mr-1"
              type="radio"
              name="radio"
              defaultChecked={ms_checked}
              onChange={() => {
                setMsChecked(true);
                setSdChecked(true);
                setFcChecked(true);
                setSiiChecked(true);
                setSrChecked(true);
                setAdChecked(false);
                setLeChecked(true);
                setLcvChecked(true);
                setLiChecked(true);
                setBChecked(true);
              }}
            />
            Manual Selection
          </label>
        </div>
      </div>
      <div className="grid grid-cols-1">
        <Menu as="div" className="relative inline-block text-left">
          <div className="w-fit">
            <Menu.Button
              disabled={ms_checked}
              className="ml-16 mt-8 inline-flex w-full px-4 py-2 text-black"
            >
              <label>
                <input
                  className="mr-2"
                  type="checkbox"
                  name="StatisticalDiscrepancies"
                  checked={
                    (sd_checked && ms_checked) || (le_checked && lcv_checked)
                  }
                  onChange={() => {
                    setSdChecked(!sd_checked);
                    setLeChecked(!(sd_checked && ms_checked));
                    setLcvChecked(!(sd_checked && ms_checked));
                  }}
                />
                {/* <ChevronDownIcon className="w-5 h-5 text-violet-200 hover:text-violet-100"
                             aria-hidden="true"/> */}
                Statistical Discrepancies
              </label>
            </Menu.Button>
          </div>
          <div>
            <Menu.Items static={ms_checked} className="origin-top-right">
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="StatisticalDiscrepancies"
                      checked={le_checked}
                      onChange={() => {
                        setLeChecked(!le_checked);
                        setSdChecked(!le_checked && lcv_checked);
                      }}
                      value="labelingError"
                    />
                    Labeling Errors
                  </label>
                </Menu.Item>
              </div>
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="StatisticalDiscrepancies"
                      value="Intra-classVariance"
                      checked={lcv_checked}
                      onChange={() => {
                        setLcvChecked(!lcv_checked);
                        setSdChecked(le_checked && !lcv_checked);
                      }}
                    />
                    Intra-class variance
                  </label>
                </Menu.Item>
              </div>
            </Menu.Items>
          </div>
        </Menu>

        <Menu as="div" className="relative inline-block text-left">
          <div className="w-fit">
            <Menu.Button
              disabled={ms_checked}
              className="ml-16 mt-8 inline-flex w-full px-4 py-2 text-black"
            >
              <label>
                <input
                  className="mr-2"
                  type="checkbox"
                  name="SizesAndRatios"
                  checked={sr_checked && ms_checked}
                  onChange={() => setSrChecked(!sr_checked)}
                />
                {/* <ChevronDownIcon
              className="w-5 h-5 text-violet-200 hover:text-violet-100"
              aria-hidden="true"
            /> */}
                Sizes and Ratios
              </label>
            </Menu.Button>
          </div>
          <div>
            <Menu.Items static={ms_checked} className="origin-top-right">
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="ObjectScaleVariation"
                      checked={sr_checked && ms_checked}
                      onChange={() => setSrChecked(!sr_checked)}
                    />
                    Object Scale Variation
                  </label>
                </Menu.Item>
              </div>
            </Menu.Items>
          </div>
        </Menu>

        <Menu as="div" className="relative inline-block text-left">
          <div className="w-fit">
            <Menu.Button
              disabled={ms_checked}
              className="ml-16 mt-8 inline-flex w-full px-4 py-2 text-black"
            >
              <label>
                <input
                  className="mr-2"
                  type="checkbox"
                  name="FileCounts"
                  checked={fc_checked && ms_checked}
                  onChange={() => setFcChecked(!fc_checked)}
                />
                {/* <ChevronDownIcon
              className="w-5 h-5 text-violet-200 hover:text-violet-100"
              aria-hidden="true"
            /> */}
                File Counts
              </label>
            </Menu.Button>
          </div>
          <div>
            <Menu.Items static={ms_checked} className="origin-top-right">
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="DatasetSize"
                      checked={fc_checked && ms_checked}
                      onChange={() => setFcChecked(!fc_checked)}
                    />
                    Dataset Size
                  </label>
                </Menu.Item>
              </div>
            </Menu.Items>
          </div>
        </Menu>

        <Menu as="div" className="relative inline-block text-left">
          <div className="w-fit">
            <Menu.Button
              disabled={ms_checked}
              className="ml-16 mt-8 inline-flex w-full px-4 py-2 text-black"
            >
              <label>
                <input
                  className="mr-2"
                  type="checkbox"
                  name="SingleImageIssues"
                  checked={
                    (sii_checked && ms_checked) || (li_checked && b_checked)
                  }
                  onChange={() => {
                    setSiiChecked(!sii_checked);
                    setLiChecked(!(sii_checked && ms_checked));
                    setBChecked(!(sii_checked && ms_checked));
                  }}
                />
                {/* <ChevronDownIcon
              className="w-5 h-5 text-violet-200 hover:text-violet-100"
              aria-hidden="true"
            /> */}
                Single Image Issues
              </label>
            </Menu.Button>
          </div>
          <div>
            <Menu.Items static={ms_checked} className="origin-top-right">
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="LightingIssue"
                      checked={li_checked}
                      onChange={() => {
                        setSiiChecked(!li_checked && b_checked);
                        setLiChecked(!li_checked);
                      }}
                    />
                    Lighting Issue
                  </label>
                </Menu.Item>
              </div>
              <div className="w-fit text-left ml-32 px-1 py-1">
                <Menu.Item>
                  <label>
                    <input
                      className="mr-2"
                      type="checkbox"
                      name="Blur"
                      checked={b_checked}
                      onChange={() => {
                        setSiiChecked(li_checked && !b_checked);
                        setBChecked(!b_checked);
                      }}
                    />
                    Blur
                  </label>
                </Menu.Item>
              </div>
            </Menu.Items>
          </div>
        </Menu>
      </div>
    </div>
  );
}
