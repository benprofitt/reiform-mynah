import { createContext } from "react";
import { DatasetList } from "./types";

const datasets: DatasetList = [
  {
    uuid: "123",
    owner_uuid: "sup",
    date_created: 1111111111,
    date_modified: 444444444444,
    dataset_name: "My OD dataset",
    versions: {
      "1": {
        entities: {
          uuid1: {
            current_label: "label1",
            original_label: "label2",
            vertices: [
              [3, 4],
              [3, 7],
              [8, 7],
              [8, 4],
            ],
          },
          uuid2: {
            current_label: "label3",
            original_label: "label1",
            vertices: [
              [10, 4],
              [7, 54],
              [7, 12],
              [4, 6],
            ],
          },
        },
        files: {
          fileid1: {
            image_version_id: "asidnadasd",
            entities: {
              label1: ["uuid1", "uuid2"],
            },
          },
          fileid2: {
            image_version_id: "asidnadasd",
            entities: {
              label2: ["uuid1"],
              label3: ["uuid2"],
            },
          },
        },
        file_entities: {
          label1: ["fileid1"],
          label2: ["fileid2"],
          label3: ["fileid3"],
        },
      },
    },
  },
  {
    uuid: "456",
    owner_uuid: "hey",
    date_created: 33333333333,
    date_modified: 66666666666,
    dataset_name: "My IC dataset",
    versions: {
      "0": {
        files: {
          "some fileid": {
            image_version_id: "0932dn0nw098n3",
            current_class: "",
            original_class: "",
            confidence_vectors: [[]],
            projections: {
              "some value": [0],
            },
            mean: [0, 0, 0],
            std_dev: [0, 0, 0],
          },
          "some fileid2": {
            image_version_id: "abcd",
            current_class: "",
            original_class: "",
            confidence_vectors: [[]],
            projections: {
              "some value": [0],
            },
            mean: [0, 0, 0],
            std_dev: [0, 0, 0],
          },
        },
        mean: [0, 0, 0],
        std_dev: [0, 0, 0],
      },
      "1": {
        files: {
          "some fileid3": {
            image_version_id: "fdsafsa",
            current_class: "",
            original_class: "",
            confidence_vectors: [[]],
            projections: {
              "some value": [0],
            },
            mean: [0, 0, 0],
            std_dev: [0, 0, 0],
          },
        },
        mean: [0, 0, 0],
        std_dev: [0, 0, 0],
      },
    },
  },
];

interface DataContextType {
  datasets: DatasetList;
}

export const DataContext = createContext<DataContextType>({
  datasets,
});

export default function DataProvider(props: {
  children: JSX.Element | JSX.Element[];
}): JSX.Element {
  const { children } = props;

  return (
    <DataContext.Provider value={{ datasets }}>{children}</DataContext.Provider>
  );
}
