import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";
import FileUploadListItem from "./file_upload_list_item";

export interface FileUploadListProps {
  files: { file: File; isFinished: boolean }[];
}

export default function FileUploadList(
  props: FileUploadListProps
): JSX.Element {
  const { files } = props;
  // The scrollable element for your list
  const parentRef = useRef<HTMLDivElement>(null);

  // The virtualizer
  const rowVirtualizer = useVirtualizer({
    count: files.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60,
  });

  return (
    <>
      {/* The scrollable element for your list */}
      <div className="w-full h-[500px] overflow-auto" ref={parentRef}>
        {/* The large inner element to hold all of the items */}
        <div
          className="relative w-full"
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
          }}
        >
          {/* Only the visible items in the virtualizer, manually positioned to be in view */}
          {rowVirtualizer.getVirtualItems().map((virtualItem) => (
            <div
              className="absolute top-0 left-0 w-full h-[60px]"
              key={virtualItem.key}
              style={{
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              <FileUploadListItem fileObj={files[virtualItem.index]} />
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
