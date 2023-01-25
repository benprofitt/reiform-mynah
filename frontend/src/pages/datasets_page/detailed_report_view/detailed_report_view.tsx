import { Redirect, useLocation } from "wouter";
import BackButton from "../../../components/back_button";
import LabelErrorBreakdown from "./label_error_breakdown";
import ImageListViewerAndScatter from "./image_list_viewer_and_scatter";
import {
  MynahICDataset,
  MynahICDatasetReport,
  MynahICProcessTaskDiagnoseClassSplittingReport,
  MynahICProcessTaskDiagnoseMislabeledImagesReport,
  MynahICProcessTaskType,
} from "../../../utils/types";
import ClassSplittingBreakdown from "./class_splitting_breakdown";
import { reportToString } from "../dataset_detail_page/reports";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";

export interface DetailedReportViewProps {
  dataId: string;
  dataset: MynahICDataset;
  basePath: string;
  reportType: MynahICProcessTaskType;
  version: string | undefined;
}

export default function DetailedReportView(props: DetailedReportViewProps) {
  const { dataId, dataset, basePath, reportType, version } = props;

  const { data, isLoading } = useQuery<MynahICDatasetReport>(
    `report-${dataId}`,
    () =>
      makeRequest<MynahICDatasetReport>("GET", `/api/v1/data/json/${dataId}`)
  );
  console.log(data)
  const [_location, setLocation] = useLocation();

  const close = () => setLocation(basePath);

  // need to check if dataId is assosiated with an actual report that we have before displaying it
  // but until i have actual reports to play with we'll have to bypass that

  const intraclassVariance =
    reportType === "ic::correct::class_splitting" ||
    reportType === "ic::diagnose::class_splitting";
  const labelError =
    reportType === "ic::correct::mislabeled_images" ||
    reportType === "ic::diagnose::mislabeled_images";

  

  if (!version || (!labelError && !intraclassVariance))
    return <Redirect to={basePath} />;

  if (!data || isLoading) return <>Loading...</>
  
  const taskMetaData = data.tasks.filter(({type}) => type==reportType)[0].metadata



  return (
    <div className="w-screen h-screen bg-white z-10 fixed top-0 left-0 flex text-black">
      {/* sidebar */}
      <div className="w-[30%] border-grey border-r-4">
        <header className="w-full border-b-2 border-black pl-[32px] py-[25px]">
          <BackButton destination={basePath} />
          <h1 className="font-black text-[28px] mt-[10px]">{dataset.dataset_name}</h1>
          <h2 className="text-grey2 font-medium text-[18px] mt-[5px]">
            {`${reportToString[reportType]} V${version}`}
          </h2>
        </header>

        {/* breakdown */}

        <div className="px-[32px] pt-[25px]">
          {intraclassVariance && (
            <ClassSplittingBreakdown
              taskMetadata={taskMetaData as MynahICProcessTaskDiagnoseClassSplittingReport}
              correctionReport={reportType === "ic::correct::class_splitting"}
            />
          )}
          {labelError && <LabelErrorBreakdown 
              taskMetadata={taskMetaData as MynahICProcessTaskDiagnoseMislabeledImagesReport} />}
        </div>
      </div>

      {/* main content */}
      <div className="flex w-[70%]">
        <ImageListViewerAndScatter reportData={data}/>
      </div>
    </div>
  );
}
