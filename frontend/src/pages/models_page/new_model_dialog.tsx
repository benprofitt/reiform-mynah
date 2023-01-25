import { Dialog } from "@headlessui/react";

export interface CreateNewModelProps {
  open: boolean;
  close: () => void;
}

export default function CreateNewModel(
  props: CreateNewModelProps
): JSX.Element {
  const { open, close } = props;
  return (
    <Dialog
      open={open}
      onClose={close}
      className="fixed inset-0 z-20 /w-full /h-full flex items-center justify-center py-[10px]"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-20" />
      <div className="w-[752px] h-fit max-h-full mx-auto flex flex-col items-center relative z-30 bg-white px-[24px]">
        <h1 className="text-[28px] w-full mt-[14px]">Create new model</h1>
        <form className="w-full">
          <div className="w-full justify-between flex">
            <label className="font-black" htmlFor="name">
              Model Name
            </label>
            <input className="w-[70%] border-grey h-[30px] border" placeholder="Enter name" />
          </div>
        </form>
      </div>
    </Dialog>
  );
}
