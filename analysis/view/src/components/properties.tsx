interface PropertyProps {
  name: string;
  value: string | number | boolean;
}

export const Property: React.FC<PropertyProps> = ({ name, value }) => {
  return (
    <div className="flex flex-row items-center gap-2">
      <span className="text-muted-foreground">{name}:</span>
      <span className="font-semibold">{typeof value === "boolean" ? (value ? "true" : "false") : value}</span>
    </div>
  );
};

export const PropertiesList: React.FC<{ properties: Record<string, string | number | boolean> }> = ({ properties }) => {
  return (
    <div className="flex flex-wrap gap-4">
      {Object.entries(properties).map(([name, value]) => (
        <Property key={name} name={name} value={value} />
      ))}
    </div>
  );
};

export const PropertiesWithImprovement: React.FC<PropertyProps & { improvement?: boolean }> = ({
  name,
  value,
  improvement,
}) => {
  return (
    <div className="flex flex-row items-center gap-2">
      <span className="text-muted-foreground">{name}:</span>
      <span className={`font-semibold ${improvement ? "text-green-500" : "text-red-500"}`}>
        {typeof value === "boolean" ? (value ? "true" : "false") : value}
      </span>
    </div>
  );
};

export const PropertiesWithImprovementList: React.FC<{
  properties: Record<string, { value: string | number | boolean; improvement?: boolean }>;
}> = ({ properties }) => {
  return (
    <div className="flex flex-wrap gap-4">
      {Object.entries(properties).map(([name, { value, improvement }]) => (
        <PropertiesWithImprovement key={name} name={name} value={value} improvement={improvement} />
      ))}
    </div>
  );
};
