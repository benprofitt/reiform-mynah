import { Menu } from "@headlessui/react";
import clsx from "clsx";
import { last } from "lodash";
import ArrowIcon from '../../../images/ArrowIcon.svg'
import RightArrowIcon from '../../../images/RightArrowIcon.svg'
import {
  CSSProperties,
  MutableRefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { FixedSizeList as List } from "react-window";

const FilterDropdown = (): JSX.Element => {
  return (
    <Menu as="div" className="relative inline-block text-left ml-auto z-10">
      <div>
        <Menu.Button className="focus:outline-none flex items-center gap-[5px]">
          Filter by
          <img src={ArrowIcon} alt="arrow" />
        </Menu.Button>
      </div>
      <Menu.Items className="absolute left-0 mt-2 w-56 origin-top-left divide-y divide-gray-100 rounded-md bg-white shadow-floating ring-1 ring-black ring-opacity-5 focus:outline-none">
        <DotMenuItem text="Date" />
        <DotMenuItem text="Size" />
      </Menu.Items>
    </Menu>
  );
};
interface DotMenuItemProps {
  src?: string;
  text: string;
}

const DotMenuItem = (props: DotMenuItemProps): JSX.Element => {
  const { src, text } = props;
  return (
    <Menu.Item>
      {({ active }) => (
        <button
          className={`${
            active ? "bg-sideBar text-white" : "text-gray-900"
          } flex w-full items-center rounded-md px-[20px] py-[15px]`}
        >
          {src && (
            <img
              src={src}
              alt={text}
              className={clsx(
                "mr-[10px] h-[20px] aspect-square",
                active && "invert"
              )}
            />
          )}
          {text}
        </button>
      )}
    </Menu.Item>
  );
};


export interface ImageListProps {
  data: Partial<Plotly.ScatterData>[];
  setPoint: (x: Plotly.Datum, y: Plotly.Datum, pointIndex: number) => void;
  last: { x: Plotly.Datum; y: Plotly.Datum; pointIndex: number } | null;
}

interface RowProps {
  index: number;
  style: CSSProperties;
  onClick: () => void;
  selected: boolean;
}

function Row(props: RowProps): JSX.Element {
  const { index, style, onClick, selected } = props;
  return (
    <div className={clsx('border-b-4 border-grey h-[80px] flex items-center px-[30px]', selected && 'bg-linkblue bg-opacity-5')} style={style} onClick={onClick}>
      <div className="h-[47px] aspect-square bg-black mr-[10px]"/> Row {index} <img src={RightArrowIcon} className='ml-auto'/>
    </div>
  );
}

export default function ImageList(props: ImageListProps): JSX.Element {
  const { data, setPoint, last } = props;
  const xList = data[0].x;
  const yList = data[0].y;

  const listRef = useRef<List | null>();

  const headerid = 'header'

  useEffect(() => {
    if (!last || !listRef.current) return;
    listRef.current.scrollToItem(last.pointIndex, 'smart');
  }, [last]);

  console.log('imagelist render')
  if (!xList || !yList) return <></>;


  return (
    <>
      <div className="h-[110px] py-[20px] px-[30px] border-b-4 border-grey" id={headerid}>
        <h3 className="mb-[10px] text-[20px] font-medium">Images</h3>
        <FilterDropdown />
      </div>
      <List
        className="no-scrollbar"
        height={window.innerHeight - 110}
        itemCount={xList.length}
        itemSize={80}
        ref={(el) => (listRef.current = el)}
        width={document.getElementById(headerid)?.clientWidth ?? 500}
      >
        {(props) => (
          <Row
            index={props.index}
            style={props.style}
            selected={last !== null && props.index === last.pointIndex}
            onClick={() =>
              setPoint(
                xList[props.index] as Plotly.Datum,
                yList[props.index] as Plotly.Datum,
                props.index
              )
            }
          />
        )}
      </List>
    </>
  );
}
