export interface PageContainerProps {
  children: JSX.Element | JSX.Element[];
}

export default function PageContainer(props: PageContainerProps): JSX.Element {
  const { children } = props;
  return (
    <div className="w-[97vw] h-[95vh] m-auto mt-5 border-2 border-black flex flex-col">
      {children}
    </div>
  );
}
