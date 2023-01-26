import { ReactNode } from "react";

export default function HomePageLayout({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen flex-1">
      <div className="w-full flex flex-col h-full">
        <header className="w-full h-[80px] border-b border-grey1 pl-[32px] flex items-center bg-white">
          <h1 className="font-black text-[28px]">{title}</h1>
        </header>
        <main className="bg-grey w-full p-[32px] flex-1 overflow-y-scroll">
          {children}
        </main>
      </div>
    </div>
  );
}
