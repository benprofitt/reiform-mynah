
import clsx from "clsx";
import { Link } from "wouter";
import BackArrowIcon from "../images/BackArrowIcon.svg";

export interface BackButtonProps {
    className?: string,
    destination: string
}

const BackButton = ({ className, destination }: BackButtonProps) => (
    <Link to={destination} >
        <button className={clsx(className, "flex items-center text-linkblue font-bold")}>
            <img
                src={BackArrowIcon}
                alt="back arrow"
                className="mr-[2px]"
            />
            Back
        </button>
    </Link>
)

export default BackButton