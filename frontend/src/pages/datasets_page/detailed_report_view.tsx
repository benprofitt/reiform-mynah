import { MynahICDatasetReport } from "../../utils/types";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  Tooltip,
  TooltipProps
} from "recharts";
import {
  ValueType,
  NameType,
} from 'recharts/src/component/DefaultTooltipContent';

function stringToColor (str: string): string {
  var hash = 0;
  for (var i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  var colour = '#';
  for (var i = 0; i < 3; i++) {
    var value = (hash >> (i * 8)) & 0xFF;
    colour += ('00' + value.toString(16)).substr(-2);
  }
  return colour;
}

const data: MynahICDatasetReport = {
  points: [
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: 3,
      y: 1,
      class: "dog",
      original_class: "cat",
    },
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: 2.5,
      y: 1.2,
      class: "dog",
      original_class: "cat",
    },
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: 2,
      y: 2,
      class: "cat",
      original_class: "cat",
    },
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: 1.75,
      y: 1.9,
      class: "cat",
      original_class: "cat",
    },
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: 1,
      y: 1,
      class: "jessica",
      original_class: "cat",
    },
    {
      fileid: "<insert id here>",
      image_version_id: "<insert id here>",
      x: .5,
      y: .5,
      class: "jake",
      original_class: "cat",
    },
  ],
  tasks: [],
};

const CustomTooltip = ({ active, payload }: TooltipProps<ValueType, NameType>) => {
  if (active && payload) {
    const { payload: thePayload } = payload[0]
    const x = thePayload['x']
    const y = thePayload['y']
    const theClass = thePayload['class']
    return (
      <div className="custom-tooltip">
        <p className="desc">x: {x}, y: {y}, class: {theClass}</p>
      </div>
    );
  }

  return null;
};

export interface DetailedReportViewProps {
  dataId: string;
  close: () => void;
}

export default function DetailedReportView(props: DetailedReportViewProps) {
  const { dataId, close } = props;

  return (
    <div className="w-screen h-screen bg-white z-10 fixed top-0 left-0 flex text-black">
      {/* sidebar */}
      <div className="w-[30%]">
        {/* sidebar header */}
        <div>
          <button className='text-linkblue' onClick={close}>back</button>
          <h2>title</h2>
          <h3>subtitle</h3>
        </div>
        {/* sidebar breakdown */}
        <div>
          <h4>Breakdown</h4>
          <h5>Table graph chart</h5>
        </div>
      </div>
      <div className="flex w-[70%]">
        {/* file list */}
        <div className="w-[30%]">images...</div>
        {/* picture and graph */}
        <div className="w-[70%]">
          <div className="h-[40%]">selected image</div>
          {/* <div className="p-[10px]"> */}
            <ResponsiveContainer width="100%" height="60%">
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: -20}}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" />
                <YAxis type="number" dataKey="y" />
          <Tooltip content={<CustomTooltip />} />
                <Scatter data={data.points} onClick={(e) => console.log(e)}>
                  {data.points.map((entry, ix) => 
                    <Cell key={ix} fill={stringToColor(entry.class)} />
                  )}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          {/* </div> */}
        </div>
      </div>
    </div>
  );
}
