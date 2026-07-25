import InputBar from "./components/InputBar";
import { useState } from "react";
import { askQuestion } from "./api/askApi";
import { toast } from "react-hot-toast";
import ThemeToggle from "./components/ThemeToggle";

export default function Demo() {
  const [isSending, setIsSending] = useState(false);
  const [getAns, setAns] = useState(null);

  const handleSend = async (question) => {
    setIsSending(true);
    try {
      const data = await askQuestion("1001", question);
      setAns(data); // 'data' holds { answer, citations, answer_found }
    } catch (err) {
      console.error(err);
      toast.error("Error to get answer");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="p-6">
      <h2>Here is demo</h2>
      <ThemeToggle/>
      <InputBar onSend={handleSend} isLoading={isSending} />

      {/* FIXED: Using getAns instead of data */}
      {getAns && (
        <div className="mt-4 p-4 border rounded bg-gray-50">
          <strong>Answer:</strong>
          <p>{getAns.answer}</p>
          
          {/* Pro tip: Since your API returns citations, you can loop through them too! */}
          {getAns.citations && getAns.citations.length > 0 && (
            <div className="mt-2 text-sm text-gray-500">
              <strong>Citations:</strong> {getAns.citations.length} sources found
            </div>
          )}
        </div>
      )}
    </div>
  );
}
