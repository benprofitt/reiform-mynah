import DatasetList from "./dataset_list";

export default function DatasetListPage(): JSX.Element {
    
  return (
    <div className="flex h-screen flex-1">
      <div className="w-full flex flex-col h-full">
        <header className="w-full h-[80px] border-b border-grey1 pl-[32px] flex items-center bg-white">
          <h1 className="font-black text-[28px]">Datasets</h1>
        </header>
        <main className="bg-grey w-full p-[32px] flex-1 overflow-y-scroll no-scrollbar">
          <DatasetList />
        </main>
      </div>
    </div>
  );
}
