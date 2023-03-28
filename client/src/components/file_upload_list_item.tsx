import clsx from "clsx";
import { useEffect, useState } from "react";

export interface FileUploadListItemProps {
  fileObj: { file: File; isFinished: boolean };
}

const blankSrc =
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

export default function FileUploadListItem(
  props: FileUploadListItemProps
): JSX.Element {
  const { fileObj } = props;
  const { file, isFinished } = fileObj;
  const { name } = file;
  const [src, setSrc] = useState(blankSrc);
  useEffect(() => {
    setSrc(URL.createObjectURL(file));
    return () => {
        setSrc(blankSrc)
        URL.revokeObjectURL(src)
    };
  }, [file]);
  return (
    <li className="flex w-full h-full items-center border-b border-grey1">
      <img
        className="h-[40px] aspect-square mr-[10px]"
        src={src}
        alt={`Preview for ${name}`}
      />
      {name}
      <div
        className={clsx(
          "animate-spin border-t border-b border-l border-sidebarSelected h-[24px] aspect-square rounded-full ml-auto mr-[10px] my-auto",
          isFinished && "hidden"
        )}
      />
    </li>
  );
}
