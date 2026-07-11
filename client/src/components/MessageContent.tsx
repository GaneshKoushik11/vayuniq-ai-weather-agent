import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChartComponent from './ChartComponent';

export default function MessageContent({ content }: { content: string }) {
    if (content.startsWith("{ \"type\": \"chart\"")) {
        const payload = JSON.parse(content);
        return <ChartComponent data={payload.data} />;
    }

    return (
        <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
                // 1. Wrap the table in a responsive, scrolling container block
                table: ({ node, ...props }) => (
                    <div className="w-full overflow-x-auto my-4 rounded-lg border border-gray-200 dark:border-gray-800 [scrollbar-width:thin]">
                        <table className="w-full border-collapse text-sm text-left text-gray-700 dark:text-gray-300" {...props} />
                    </div>
                ),
                thead: ({ node, ...props }) => (
                    <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-600 border-b border-gray-200 dark:bg-gray-900/50 dark:text-gray-400 dark:border-gray-800" {...props} />
                ),
                th: ({ node, ...props }) => (
                    <th className="px-4 py-3 font-semibold whitespace-nowrap" {...props} />
                ),
                td: ({ node, ...props }) => (
                    // 2. break-words allows text to wrap onto new lines on small screens instead of overflowing horizontally
                    <td className="px-4 py-3 border-b border-gray-100 dark:border-gray-800 max-w-[200px] break-words" {...props} />
                ),
                tr: ({ node, ...props }) => (
                    <tr className="odd:bg-white even:bg-gray-50/50 hover:bg-gray-100/70 transition-colors dark:odd:bg-transparent dark:even:bg-gray-900/20 dark:hover:bg-gray-800/50" {...props} />
                ),
            }}
        >
            {content}
        </ReactMarkdown>
    );
}