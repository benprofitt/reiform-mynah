import { Dialog } from "@headlessui/react";
import { useQuery } from "react-query";
import { useLocation } from "wouter";
import makeRequest from "../../../utils/apiFetch";
import { MynahFile, MynahICDataset } from "../../../utils/types";
import Image from "../../../components/Image";

const MetaDetail = (props: { title: string; value: string }): JSX.Element => {
  const { title, value } = props;
  return (
    <div className="grid grid-cols-2 ml-[24px]">
      <h4 className="font-bold">{title}:</h4>
      <p>{value}</p>
    </div>
  );
};

export default function DetailedFileView(props: {
  versions: MynahICDataset["versions"];
  fileId: string | undefined;
  basePath: string;
  fileData: MynahFile | undefined;
}): JSX.Element {
  const [_location, setLocation] = useLocation();
  const { versions, fileId, basePath, fileData } = props;
  if (fileId == undefined || fileData == undefined) return <></>
//   if (fileData === undefined) return <></>
//   const query = `/api/v1/file/list?fileid=${fileId ?? ""}`;
//   const { error, isLoading, data } = useQuery(`datasetFile-${fileId}`, () =>
//     makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
//   );
//   if (fileId === undefined) return <></>;
//   if (data === undefined || isLoading || error) return <></>;
//   const fileData = data[fileId];
  const close = () => setLocation(basePath);

  return (
    <Dialog
      onClose={close}
      open={true}
      className="fixed inset-0 w-full h-full"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute top-0 left-0 opacity-20 z-0" />
      <main className="bg-white z-10 h-full w-fit right-0 top-0 absolute px-[24px] max-w-[calc(100%-30px)]">
        <button
          className="absolute top-[15px] right-[20px] text-[40px] leading-none"
          onClick={close}
        >
          x
        </button>
        <h1 className="py-[24px] font-black text-[28px]">{fileData.name}</h1>
        {Object.keys(versions).map((version) => {
          const file = versions[version].files[fileId];
          if (file === undefined) return <></>;
          const { image_version_id, current_class, original_class } = file; // img_version_id should replace latest in the image src
          return (
            <div className="flex border-grey1 border-2 h-[350px]" key={version}>
              <Image
                src={`/api/v1/file/${fileId}/${image_version_id}`}
                className="max-w-[min(60%,700px)] object-contain"
              />
              <div className="w-[376px]">
                <h3 className="text-[20px] p-[24px]">Version: {version}</h3>
                <div className="grid gap-[10px]">
                  <MetaDetail title="Class" value={current_class} />
                  <MetaDetail title="Original Class" value={original_class} />
                </div>
              </div>
            </div>
          );
        })}
      </main>
    </Dialog>
  );
}
