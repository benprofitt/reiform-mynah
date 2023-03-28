import { useQuery } from "@tanstack/react-query";
import { MynahDataset, Paginated } from "../types";
import makeRequest from "../utils/apiFetch";
import { ReactComponent as PlaceHolderImg } from "../assets/LAXPlaceholder.svg";

import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useState } from "react";
import getDate from "../utils/date";

const columnHelper = createColumnHelper<Omit<MynahDataset, "dataset_id">>();

const columns = [
  columnHelper.accessor("dataset_name", {
    cell: (info) => info.getValue(),
    header: () => <span>Name</span>,
  }),
  columnHelper.accessor("date_created", {
    cell: (info) => getDate(info.getValue()),
    header: () => <span>Created</span>,
  }),
  columnHelper.accessor("date_modified", {
    cell: (info) => getDate(info.getValue()),
    header: () => <span>Last modified</span>,
  }),
  columnHelper.accessor("dataset_type", {
    cell: (info) => <span className="capitalize">{info.getValue().replace('_', ' ')}</span>,
    header: () => <span>Dataset Type</span>,
  }),
];

export default function DatasetList(): JSX.Element {
  const { data, isLoading, isError } = useQuery(["datasets"], () =>
    makeRequest<Paginated<MynahDataset>>("GET", "/api/v2/dataset/list")
  );
  const table = useReactTable({
    data: data ? data.contents : [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  });
  if (data === undefined || isLoading) {
    return <span>Loading...</span>;
  }

  if (isError) {
    return <span>Error retrieving datasets</span>;
  }
  return (
    <>
    <h4>{data.total} total datasets</h4>
    <table className="w-full border-separate border-spacing-y-[10px]">
      <thead className="text-left">
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            <th></th>
            {headerGroup.headers.map((header) => (
              <th key={header.id}>
                {header.isPlaceholder
                  ? null
                  : flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id} className="hover:shadow-floating cursor-pointer bg-white h-[70px] border border-grey4 rounded-md text-[18px]">
            <td><PlaceHolderImg /></td>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    </>
    
  );
}
