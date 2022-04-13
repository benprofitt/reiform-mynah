import { MynahICDataset } from "../utils/types";

export interface ChooseDatasetProps {
  datasets: MynahICDataset[];
  selectedDatasets: string[];
  setSelectedDatasets: React.Dispatch<React.SetStateAction<string[]>>;
}

export default function ChooseDataset(props: ChooseDatasetProps): JSX.Element {
  const { datasets, selectedDatasets, setSelectedDatasets } = props;

  return (
    <div className="mt-8 h-96 w-96 border-2 border-black">
      <h3 className="h-8 text-center border-b-2 border-black">
        Choose Dataset
      </h3>
      <div className="h-10 border-b-2 border-black flex flex-row justify-between px-10 w-full items-center">
        <div className="h-6 w-16 text-center border-2 border-black">Name</div>
        <div className="text-center w-32 h-6 border-2 border-black">
          Date Created
        </div>
      </div>
      <form>
        {datasets.map(({ dataset_name, uuid }) => (
          <div className="h-12 border-b-2 border-black flex" key={uuid}>
            <div className="h-12 w-12 border-r-2 border-black">
              <div className="form-check">
                <input
                  className="ml-4 mt-4 form-check-input appearance-none rounded-full h-4 w-4 border border-black checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
                  type="checkbox"
                  name="source"
                  checked={selectedDatasets.includes(uuid)}
                  onChange={() => {
                    // selectedDatasets with this uuid removed
                    const newSelected = selectedDatasets.filter(
                      (id) => id !== uuid
                    );
                    // if nothing got removed then add it
                    if (newSelected.length === selectedDatasets.length)
                      newSelected.push(uuid);
                    setSelectedDatasets(newSelected);
                  }}
                />
              </div>
            </div>
            <div className="grow text-left pl-4 pt-2">{dataset_name}</div>
          </div>
        ))}
      </form>
    </div>
  );
}
