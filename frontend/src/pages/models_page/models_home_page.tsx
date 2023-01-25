import { Link } from "wouter";
import HomePageLayout from "../../components/home_page_layout";
import EllipsisMenu from "../../components/ellipsis_menu";
import FilterDropdown from "../../components/filter_dropdown";
import MenuItem from "../../components/menu_item";

interface MyModel {
  name: string;
  stage: string;
  datasetsUsed: string[];
  created: string;
  lastTrained: string;
  id: string;
}

const models: MyModel[] = [
  {
    name: "XRay Classifier",
    stage: "Processing",
    datasetsUsed: ["Pneumonia", "COVID", "Broken Arm"],
    created: "Mar. 12 2022",
    lastTrained: "Today",
    id: "456",
  },
  {
    name: "Rash Classifier",
    stage: "Complete",
    datasetsUsed: ["RoadRash 1ms", "Poison Oak"],
    created: "Mar. 12 2022",
    lastTrained: "Apr.4 2022",
    id: "123",
  },
];

const ModelListItem = ({ model }: { model: MyModel }): JSX.Element => {
  const { name, stage, datasetsUsed, lastTrained, created, id } = model;
  return (
    <Link to={`/models/${id}`}>
      <div className="hover:shadow-floating cursor-pointer bg-white h-[70px] border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative">
        <h6 className="w-[20%] font-bold pl-[20px] text-black">{name}</h6>
        <h6 className="w-[20%]">{stage}</h6>
        <h6 className="w-[20%]">{datasetsUsed.join(', ')}</h6>
        <h6 className="w-[20%]">{created}</h6>
        <h6 className="w-[20%]">{lastTrained}</h6>
        <EllipsisMenu>
          <MenuItem text="..." />
        </EllipsisMenu>
      </div>
    </Link>
  );
};

const MainModelsContent = (): JSX.Element => {
  const numModels = models.length;
  const modelsCount =
    numModels === 1 ? "One Model" : `${numModels} total models`;
  return (
    <div className="text-grey2">
      <div className="flex justify-between">
        <h3>{modelsCount}</h3>
        <FilterDropdown />
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[20%]">Name</h5>
        <h5 className="w-[20%]">Stage</h5>
        <h5 className="w-[20%]">Datasets Used</h5>
        <h5 className="w-[20%]">Created</h5>
        <h5 className="w-[20%]">Last Modified</h5>
      </div>
      <div>
        {models.map((model, ix) => (
          <ModelListItem model={model} key={ix} />
        ))}
      </div>
    </div>
  );
};

export default function ModelsHomePage(): JSX.Element {
  return (
    <HomePageLayout title="Models">
      <MainModelsContent />
    </HomePageLayout>
  );
}
