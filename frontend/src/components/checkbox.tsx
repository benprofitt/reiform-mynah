import react from "react";
import { Link } from "wouter";

export interface CheckBoxesProps {
    Label: string;
}

export default function CheckBoxes(props: CheckBoxesProps): JSX.Element {
    const { Label } = props;
    return (
        <div>
            <div className="form-check">
                <input className="form-check-input appearance-none ml-4 h-4 w-4 border border-gray-300 rounded-sm bg-white checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 mt-1 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
                    type="checkbox"
                    id="flexCheckDefault" />
                <label className="form-check-label inline-block text-gray-800">
                    {Label}
                </label>
            </div>
        </div>
    );
}