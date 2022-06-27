import clsx from "clsx";

export interface NextButtonProps {
  onClick: () => void;
  text: string;
  active?: boolean;
}

export default function NextButton(props: NextButtonProps): JSX.Element {
  const { onClick, text, active = true } = props;
  return (
    <button
      className={clsx(
        "absolute -top-16 right-8 text-center text-black h-12 w-56 flex items-center justify-center rounded",
        active ? "bg-green-300" : "bg-slate-300 pointer-events-none"
      )}
      onClick={onClick}
    >
      {text}
    </button>
  );
}
