import { Menu } from "@headlessui/react";
import { useState } from "react";
import x from "../../../images/x-circle.svg";

const datasets = ["dataset 1", "2nd dataset", "ooo a third"];
const inputs = [
  "Max epoches",
  "Min epoches",
  "Train/Test split",
  "Model architecture",
];

function ConfigurationSection({
  title,
  children,
}: {
  title: string;
  children?: JSX.Element;
}): JSX.Element {
  return (
    <div className="grid grid-flow-col grid-cols-3 lg:grid-cols-5">
      <h3 className="font-bold">{title}</h3>
      <div className="col-span-2">{children}</div>
    </div>
  );
}

function FormItem({
  label,
  children,
}: {
  label: string;
  children?: JSX.Element;
}): JSX.Element {
  return (
    <div className="grid grid-flow-col grid-cols-7">
      <h3 className="col-span-3">{label}</h3>
      <div className="col-span-4">{children}</div>
    </div>
  );
}

export default function Configuration(): JSX.Element {
    const [selections, setSelections] = useState<string[]>([])
  return (
    <div>Configuration??</div>
    // <div>
    //   <ConfigurationSection title="Datasets">
    //     <div>
    //       <Menu>
    //         <Menu.Button>Select datasets</Menu.Button>
    //         <Menu.Items>
    //           {datasets.map((name, ix) => (
    //             <Menu.Item key={ix}>
    //               <div>{name}</div>
    //             </Menu.Item>
    //           ))}
    //         </Menu.Items>
    //       </Menu>
    //       <p>h5</p>
    //       <div>
    //         <p>Set 1 lorem</p>
    //         <img src={x} />
    //       </div>
    //     </div>
    //   </ConfigurationSection>
    //   <ConfigurationSection title="Settings">
    //     <div className="grid grid-cols-1 gap-[10px]">
    //       <FormItem label="Max Epochs">
    //         <input className="w-full" />
    //       </FormItem>
    //       <FormItem label="Min Epochs">
    //         <input className="w-full"/>
    //       </FormItem>
    //       <FormItem label="Train/Test Split">
    //         <div className="grid grid-cols-2 gap-[10px]">
    //           <input /> <input />
    //         </div>
    //       </FormItem>
    //       <FormItem label="Model Architecture">
    //         <input className="w-full"/>
    //       </FormItem>
    //     </div>
    //   </ConfigurationSection>
    // </div>
  );
}
