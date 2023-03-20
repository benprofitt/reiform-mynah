import { useQuery } from "@tanstack/react-query";
import { MynahDataset, Paginated } from "../types";
import makeRequest from "../utils/apiFetch";
import DatasetListItem from "./dataset_list_item";

export default function DatasetList(): JSX.Element {
  const { data, isLoading, isError } = useQuery(["datasets"], () =>
    makeRequest<Paginated<MynahDataset>>("GET", "/api/v2/dataset/list")
  );

  if (data === undefined || isLoading) {
    return <span>Loading...</span>;
  }

  if (isError) {
    return <span>Error retrieving datasets</span>;
  }
  return (
    <ul>
      {data.contents.map((MynahDataset, ix) => (
        <DatasetListItem key={ix} dataset={MynahDataset} />
      ))}
    </ul>
  );
}
