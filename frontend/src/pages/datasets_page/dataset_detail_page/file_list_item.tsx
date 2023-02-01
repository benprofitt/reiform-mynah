import { MynahFile, MynahICFile } from "../../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import Image from "../../../components/Image";
import getDate from "../../../utils/date";
import { Link, useLocation } from "wouter";
import { CSSProperties, memo, useRef } from "react";
import EllipsisMenu from "../../../components/ellipsis_menu";

function FileListItem
    (props: {
      index: number;
      style: CSSProperties;
      version: string;
      fileId: string;
      file: MynahICFile;
      basePath: string;
      fileData: MynahFile;
    }): JSX.Element {
      const { version, fileId, file, basePath, index, style, fileData } = props;
  
      const { image_version_id } = file;
      const imgClass = "current_class" in file ? file.current_class : "OD class";
  
    //   const query = `/api/v1/file/list?fileid=${fileId}`;
    //   const { error, isLoading, data } = useQuery(`datasetFile-${fileId}`, () =>
    //     makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
    //   );
    //   console.log(data);
  
    //   if (!data || isLoading || error)
    //     return <div className="w-full bg-white">getting file...</div>;
    //   const fileData = data[fileId];
      const { date_created, name: fileName } = fileData;
  
      return (
        <div style={style}>
          <Link
            className="w-full hover:shadow-floating cursor-pointer bg-white border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative text-left h-[72px]"
            to={`${basePath}/${fileId}`}
          >
            <h6 className="w-[40%] font-black text-black flex items-center">
              <Image
                src={`/api/v1/file/${fileId}/${image_version_id}`}
                className="w-[9%] min-w-[70px] aspect-square mr-[1.5%] object-cover"
              />
              {fileName}
            </h6>
            <h6 className="w-[20%]">{getDate(date_created)}</h6>
            <h6 className="w-[20%]">{imgClass}</h6>
            <h6 className="w-[20%]">{version}</h6>
            <EllipsisMenu />
          </Link>
        </div>
      );
    }

export default memo(FileListItem)
  