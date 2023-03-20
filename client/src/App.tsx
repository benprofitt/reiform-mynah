import Sidebar from "./components/sidebar";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import DatasetListPage from "./components/dataset_list_page";

const queryClient = new QueryClient();
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex">
        <Sidebar />
        <DatasetListPage />
      </div>
    </QueryClientProvider>
  );
}

export default App;
