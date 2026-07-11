import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChartComponent from './ChartComponent';

export default function MessageContent({ content }: { content: string }) {
    if (content.startsWith("{ \"type\": \"chart\"")) {
        const payload = JSON.parse(content);
        return <ChartComponent data={payload.data} />;
    }

    return (
        /* The container forces block display and handles horizontal overflow without breaking the parent width */
        <div className="block w-full overflow-x-auto my-6 rounded-xl border border-gray-200 shadow-sm dark:border-gray-800">
            <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                components={{
                    table: ({ node, ...props }) => (
                        <table className="w-full border-collapse text-sm text-left text-gray-700 dark:text-gray-300" {...props} />
                    ),
                    thead: ({ node, ...props }) => (
                        <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-600 border-b border-gray-200 dark:bg-gray-900/50 dark:text-gray-400 dark:border-gray-800" {...props} />
                    ),
                    th: ({ node, ...props }) => (
                        <th className="px-6 py-3.5 font-semibold" {...props} />
                    ),
                    td: ({ node, ...props }) => (
                        /* max-w-xs + truncate prevents cells from expanding infinitely while keeping layout tight */
                        <td className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 max-w-xs truncate" {...props} />
                    ),
                    tr: ({ node, ...props }) => (
                        /* Zebra striping makes large amounts of tabular data vastly easier to scan */
                        <tr className="odd:bg-white even:bg-gray-50/50 hover:bg-gray-100/70 transition-colors dark:odd:bg-transparent dark:even:bg-gray-900/20 dark:hover:bg-gray-800/50" {...props} />
                    ),
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
}