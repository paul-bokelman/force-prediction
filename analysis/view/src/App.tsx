import type { CandidateReport } from "@/types/data.types";
import * as React from "react";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { PropertiesList, Divider, PropertiesWithImprovementList } from "@/components";

// import all .json files under src/data/candidates at build time
const modules = import.meta.glob("./data/candidates/*.json", { eager: true });

// build a map of filename -> parsed JSON
const candidateMap: Record<string, CandidateReport> = Object.fromEntries(
  Object.entries(modules).map(([path, mod]) => {
    const fileName = path.split("/").pop()!;
    return [fileName, mod as CandidateReport];
  })
);

const candidateFiles = Object.keys(candidateMap);

function App() {
  const [selectedFile, setSelectedFile] = React.useState<string | null>(null);
  const [data, setData] = React.useState<CandidateReport | null>(null);

  React.useEffect(() => {
    if (candidateFiles.length > 0) {
      setSelectedFile(candidateFiles[0]);
    }
  }, []);

  React.useEffect(() => {
    if (!selectedFile) return;
    setData(candidateMap[selectedFile]);
  }, [selectedFile]);

  if (!data) {
    return (
      <div className="flex w-screen h-screen items-center justify-center">
        <p className="text-sm">No initial data loaded.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-12">
      <div className="flex flex-row items-center gap-4">
        <Label htmlFor="file-select" className="text-sm">
          Select a candidate:
        </Label>

        <Select value={selectedFile ?? ""} onValueChange={setSelectedFile}>
          <SelectTrigger>
            <SelectValue placeholder="-- Select file --" />
          </SelectTrigger>
          <SelectContent>
            {candidateFiles.map((file) => (
              <SelectItem key={file} value={file}>
                {file.replace(/\.json$/, "")}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <Divider />
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-bold">{data.candidate.hash}</h1>
        <PropertiesList
          properties={{
            identifier: data.candidate.identifier,
            architecture: data.candidate.architecture,
            version: data.candidate.version,
          }}
        />
        <span className="uppercase text-muted-foreground text-sm italic">Preprocessing Hyperparameters</span>
        <PropertiesList properties={data.candidate.hyperparameters.preprocessing} />
        <span className="uppercase text-muted-foreground text-sm italic">Training Hyperparameters</span>
        <PropertiesList properties={data.candidate.hyperparameters.training} />
      </div>
      <Divider />
      <PropertiesWithImprovementList
        properties={Object.fromEntries(
          Object.entries(data.metrics).map(([key, { baseline, candidate, improvement }]) => [
            key,
            { value: `${baseline} → ${candidate}`, improvement },
          ])
        )}
      />
      <Divider />
      <img src={`data:image/png;base64,${data.plots.training_history}`} className="w-full h-auto rounded-md" />
      <Divider />
      <div className="flex flex-col gap-2">
        {Object.entries(data.plots.predictions).map(([, value]) => (
          <img key={value} src={`data:image/png;base64,${value}`} className="w-full h-auto rounded-md" />
        ))}
      </div>
      <Divider />
      <div className="flex flex-col gap-2">
        {Object.entries(data.plots.metrics).map(([, value]) => (
          <img key={value} src={`data:image/png;base64,${value}`} className="w-full h-auto rounded-md" />
        ))}
      </div>
    </div>
  );
}

export default App;
